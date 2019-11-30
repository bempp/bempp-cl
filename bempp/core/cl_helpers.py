"""Interface to an OpenCL device."""
# pylint: disable=no-member

import collections as _collections
import contextlib as _contextlib
import os as _os

import numpy as _np
import pyopencl as _cl

_MAX_MEM_ALLOC_SIZE = _cl.device_info.MAX_MEM_ALLOC_SIZE
_GLOBAL_MEM_SIZE = _cl.device_info.GLOBAL_MEM_SIZE


_CURRENT_PATH = _os.path.dirname(_os.path.realpath(__file__))
_INCLUDE_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/include"))
_KERNEL_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/kernels"))

_DEFAULT_DEVICE = None
_DEFAULT_CONTEXT = None

TypeContainer = _collections.namedtuple("TypeContainer", "real complex opencl")


class Context(object):
    """Administrate an OpenCL Context."""

    def __init__(self, cl_context):
        """Initialize with a CL context."""

        self._cl_context = cl_context
        self._devices = None

        self._populate_devices()

    @property
    def cl_context(self):
        """Return OpenCL Context."""
        return self._cl_context

    @property
    def devices(self):
        """Return all devices in context."""
        return self._devices

    def _populate_devices(self):
        """Enumerate device interfaces."""
        self._devices = []
        for device in self.cl_context.devices:
            self.devices.append(DeviceInterface(self, device))


class DeviceInterface(object):
    """Provides an easy interface to an OpenCL device."""

    def __init__(self, context, cl_device):
        """Initialize with a context and a device."""

        self._context = context
        self._cl_device = cl_device

        self._queue = _cl.CommandQueue(
            context.cl_context,
            device=cl_device,
            properties=_cl.command_queue_properties.PROFILING_ENABLE,
        )

        # For now use the same queue for data and kernels
        # If this is ever changed be careful with the release
        # operation in the memory copy. It seems only blocking within
        # the same queue.

    @property
    def context(self):
        """Return context."""
        return self._context

    @property
    def cl_device(self):
        """Return reference to device."""
        return self._cl_device

    @property
    def data_queue(self):
        """Data queue."""
        return self._queue

    @property
    def kernel_queue(self):
        """Kernel queue."""
        return self._queue

    @property
    def queue(self):
        """Device queue."""
        return self._queue

    @property
    def global_mem_size(self):
        """Return global memory size."""
        return self._device_property(_GLOBAL_MEM_SIZE)

    @property
    def max_mem_alloc_size(self):
        """Maximum allowed object size in global memory."""
        return self._device_property(_MAX_MEM_ALLOC_SIZE)

    @property
    def name(self):
        """Return device name."""
        return self.cl_device.name

    @property
    def type(self):
        """Return 'gpu', 'cpu' or 'unknown'."""
        if self.cl_device.type == _cl.device_type.CPU:
            return "cpu"
        if self.cl_device.type == _cl.device_type.GPU:
            return "gpu"
        return "unknown"

    def native_vector_width(self, precision):
        """Return native vector width for 'single' and 'double'."""
        if precision == "single":
            return self.cl_device.native_vector_width_float
        if precision == "double":
            return self.cl_device.native_vector_width_double
        raise ValueError("precision must be one of 'single' or 'double'")

    def _device_property(self, name):
        """Return a device property."""
        return self.cl_device.get_info(name)

    def __repr__(self):
        """String representation of interface."""
        return "{0}: ".format(self.__class__) + self.name


KernelSource = _collections.namedtuple("KernelSource", "str options")


def kernel_source_from_identifier(identifier, kernel_parameters):
    """
    Initialize a kernel source object from an identifier.

    The kernel source file associated with an identifier is
    obtained by appending .cl to the identifier string.

    """
    file_name = identifier + ".cl"
    return kernel_source_from_file(file_name, kernel_parameters)


def kernel_source_from_file(file_name, kernel_parameters):
    """
    Initialize a KernelSource object from a .cl file.

    All files are assumed to reside in the default Bempp
    kernel source directory.

    """
    kernel_file = _os.path.join(_KERNEL_PATH, file_name)
    return KernelSource(open(kernel_file).read(), kernel_parameters)


class Kernel(object):
    """Wrapper for a compiled kernel."""

    def __init__(self, kernel_source, context, precision):
        """
        Initialize kernel from KernelSource object and device.

        """
        self._kernel_source = kernel_source
        self._context = context
        self._precision = precision
        self._prg = self._build(context)
        self._kernel_name = self._prg.kernel_names
        self._compiled_kernel = None

    def _build(self, context):
        """Build the OpenCL kernel instance."""

        options = _get_kernel_compile_options_from_parameters(
            self.kernel_source.options, self._precision
        )

        # Compile and return the program

        prg = _cl.Program(context.cl_context, self.kernel_source.str).build(
            options=options
        )

        return prg

    @property
    def implementation(self):
        """Return the compiled CL Kernel."""
        if self._compiled_kernel is None:
            self._compiled_kernel = getattr(self._prg, self._kernel_name)
        return self._compiled_kernel

    @property
    def kernel_source(self):
        """Return the source for the kernel."""
        return self._kernel_source

    @property
    def context(self):
        """Return device."""
        return self._context

    def run(
        self,
        device_interface,
        global_size,
        local_size,
        *args,
        global_offset=None,
        wait_for=None,
        g_times_l=False,
    ):
        """
        Runs the kernel and returns an event object.

        The parameters are the same as for
        the corresponding PyOpenCL kernel call command.

        The method returns a cl_helpers.Event object.

        """
        kernel_args = [
            arg.buffer if isinstance(arg, DeviceBuffer) else arg for arg in args
        ]

        # Disable not callable error for this method.
        # pylint: disable=E1102
        cl_event = self.implementation(
            device_interface.kernel_queue,
            global_size,
            local_size,
            *kernel_args,
            global_offset=global_offset,
            wait_for=wait_for,
            g_times_l=g_times_l,
        )

        return Event(cl_event)

    def optimal_workgroup_multiple(self, device_interface):
        """Returns optimal workgroup multiplier."""
        return self.implementation.get_work_group_info(
            _cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            device_interface.cl_device,
        )

    def __del__(self):
        """
        Need to clean up the kernel before prg is cleaned up.

        This seems to be an issue with resource handling in PyOpenCL.

        """
        if self._compiled_kernel is not None:
            self._compiled_kernel = None
        self._prg = None


def _get_kernel_compile_options_from_parameters(parameters, precision):
    """Create compiler options from parameters and precision."""
    import numbers

    if precision == "single":
        literal = "f"
    else:
        literal = ""

    if parameters is None:
        parameters = dict()

    options = []
    for key, value in parameters.items():
        if value is None:
            options += ["-D", "{0}".format(key)]
        else:
            if isinstance(value, numbers.Real) and not isinstance(
                value, numbers.Integral
            ):
                value_string = str(value) + literal
            else:
                value_string = str(value)
            options += ["-D", "{0}={1}".format(key, value_string)]

    options += ["-I", _INCLUDE_PATH]

    # Add precision flag

    if precision == "single":
        val = 0
    else:
        val = 1

    options.append("-DPRECISION={0}".format(val))

    return options


class DeviceBuffer(object):
    """Interface for device buffers."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        shape,
        dtype,
        context,
        access_mode="read_write",
        order="C",
        alloc_host_memory=False,
    ):
        """
        Initialize a new buffer.

        Parameters
        ----------
        shape : tuple
            The shape of the buffer
        dtype : object
            The Numpy type of the buffer
        context : Context
            Context in which to create the buffer
        access_mode : string
            One of 'read_only', 'write_only' or 'read_write' to
            denote if the kernel only reads, only writes
            or can do both.
        order : string
            'C' for C-stype ordering, 'F' for Fortran
            type ordering
        alloc_host_memory: boolean
            Allocate host memory to allow fast data transfers
            between device buffer and host.

        """

        self._shape = shape
        self._dtype = _np.dtype(dtype)
        self._access_mode = access_mode
        self._order = order
        self._context = context
        self._alloc_host_memory = alloc_host_memory

        self._buffer = None
        self._initialize_buffer()

    @property
    def shape(self):
        """Return the shape."""
        return self._shape

    @property
    def dtype(self):
        """Return type."""
        return self._dtype

    @property
    def access_mode(self):
        """Return access mode."""
        return self._access_mode

    @property
    def alloc_host_memory(self):
        """Return true of host memory has been allocated for buffer."""
        return self._alloc_host_memory

    @property
    def order(self):
        """Return storage order."""
        return self._order

    @property
    def context(self):
        """Return the context."""
        return self._context

    @property
    def buffer(self):
        """Return reference to buffer."""
        return self._buffer

    @_contextlib.contextmanager
    def host_array(self, device_interface, access_type, wait_for=None):
        """
        Return context manager for array access on host.

        This method creates a context manager, which
        in its as statement contains an ndarray object.
        Upon entering the context manager the buffer is mapped
        to the host. On exiting the context manager the buffer
        is unmapped.

        Parameters
        ----------
        device_interface : DeviceInterface
            The device interface object from which to
            map the buffer.
        access_type : string
            One of 'read' or 'write'
        wait_for : list
            A list of events to wait for.

        """
        if access_type == "read":
            flag = _cl.map_flags.READ
        elif access_type == "write":
            flag = _cl.map_flags.WRITE
        else:
            raise ValueError("access_mode must be one of 'read' or 'write'.")

        wait_list = None
        if wait_for is not None:
            wait_list = [event.implementation for event in wait_for]

        array, event = _cl.enqueue_map_buffer(
            device_interface.data_queue,
            self.buffer,
            flag,
            0,
            shape=self.shape,
            dtype=self.dtype,
            order=self.order,
            wait_for=wait_list,
        )
        event.wait()
        yield array
        array.base.release()

    def get_host_copy(self, device_interface):
        """
        Returns a host copy of the buffer.

        This method is a convenience routine that incurs one
        or two copy operations depending on device characteristics.
        If copy is expensive use the more low-level host_array
        method.

        """
        with self.host_array(device_interface, "read") as array:
            result = _np.copy(array)
        return result

    def fill_buffer(self, device_interface, vec):
        """Fill a buffer with data in vec. """

        with self.host_array(device_interface, "write") as array:
            array[:] = vec

    def _get_mem_flags(self):
        """Return the correct mem flags for PyOpenCL."""
        mem_flags = _cl.mem_flags
        flags = 0

        if self.alloc_host_memory:
            flags = flags | mem_flags.ALLOC_HOST_PTR
        if self.access_mode == "read_only":
            flags = flags | mem_flags.READ_ONLY
        elif self.access_mode == "write_only":
            flags = flags | mem_flags.WRITE_ONLY
        elif self.access_mode == "read_write":
            flags = flags | mem_flags.READ_WRITE
        else:
            raise ValueError(
                "'access_mode' must be one of 'read_only'"
                + " 'write_only', or 'read_write'."
            )

        return flags

    def _initialize_buffer(self):
        """Initialize the device buffer."""
        mem_flags = self._get_mem_flags()
        nbytes = _np.prod(self.shape, dtype="int64") * self.dtype.itemsize
        self._buffer = _cl.Buffer(self.context.cl_context, mem_flags, nbytes)

    def set_zero(self, device_interface):
        """Zero out the buffer."""
        nbytes = _np.prod(self.shape, dtype="int64") * self.dtype.itemsize
        event = _cl.enqueue_fill_buffer(
            device_interface.data_queue, self._buffer, _np.int8(0), 0, nbytes
        )
        event.wait()

    @classmethod
    def from_array(
        cls, array, device_interface, dtype=None, order=None, access_mode="read_write"
    ):
        """
        Return a new device buffer from an existing numpy array.

        If dtype or order are not specified, the corresponding values
        from the array are used.

        """
        if order is None:
            if array.flags["C_CONTIGUOUS"]:
                order = "C"
            else:
                order = "F"

        if dtype is None:
            dtype = array.dtype
        else:
            dtype = _np.dtype(dtype)

        buffer = cls(
            array.shape,
            dtype,
            device_interface.context,
            access_mode=access_mode,
            order=order,
        )
        with buffer.host_array(device_interface, "write") as host_array:
            host_array[:] = array

        return buffer


class Event(object):
    """Stores event information."""

    def __init__(self, cl_event):
        """Initialize with PyOpenCL event."""
        self._cl_event = cl_event

    @property
    def cl_event(self):
        """Return PyOpenCL event."""
        return self._cl_event

    @property
    def finished(self):
        """Return true if finished, otherwise false."""
        complete = _cl.command_execution_status.complete
        return (
            self.cl_event.event_info(_cl.event_info.COMMAND_EXECUTION_STATUS)
            == complete
        )

    def wait(self):
        """Wait for event to finish."""
        self.cl_event.wait()

    def runtime(self):
        """Return the runtime in milliseconds."""
        self.wait()
        return 1e-6 * (self.cl_event.profile.end - self.cl_event.profile.start)


def wait_for_events(event_list):
    """Wait for a list of events to finish."""
    cl_events = [event.cl_event for event in event_list]
    _cl.wait_for_events(cl_events)


def get_type(precision):
    """Return a TypeContainer depending on the given precision."""
    if precision == "single":
        return TypeContainer("float32", "complex64", "float")
    if precision == "double":
        return TypeContainer("float64", "complex128", "double")
    raise ValueError("precision must be one of 'single' or 'double'")


def default_device():
    """Return the default device."""
    import bempp.api
    import os

    # pylint: disable=W0603
    global _DEFAULT_DEVICE
    global _DEFAULT_CONTEXT

    if _DEFAULT_DEVICE is None:
        if not "PYOPENCL_CTX" in os.environ:
            pair = find_cpu_driver()
            if pair is not None:
                _DEFAULT_CONTEXT = pair[0]
                _DEFAULT_DEVICE = pair[1]
                bempp.api.log(
                    f"OpenCL Device set to: {_DEFAULT_DEVICE.name}")
                return _DEFAULT_DEVICE
        context = Context(_cl.create_some_context(interactive=False))
        _DEFAULT_CONTEXT = context
        _DEFAULT_DEVICE = context.devices[0]
        bempp.api.log(f"OpenCL Device set to: {_DEFAULT_DEVICE.name}")

    return _DEFAULT_DEVICE

def find_cpu_driver():
    """Find the first available CPU OpenCL driver."""

    for platform in _cl.get_platforms():
        ctx = Context(_cl.Context(
            dev_type=_cl.device_type.ALL,
	    properties=[(_cl.context_properties.PLATFORM, platform)]))
        for device in ctx.devices:
            if device.type == 'cpu':
                return ctx, device
    return None


def default_context():
    """Return default context."""
    import bempp.api

    # pylint: disable=W0603
    global _DEFAULT_CONTEXT

    if _DEFAULT_CONTEXT is None:
        # The following sets a device and context
        default_device()

    return _DEFAULT_CONTEXT


def set_default_device(platform_index, device_index):
    """Set the default device."""
    import bempp.api

    # pylint: disable=W0603
    global _DEFAULT_DEVICE
    global _DEFAULT_CONTEXT

    platform = _cl.get_platforms()[platform_index]
    device = platform.get_devices()[device_index]
    _DEFAULT_CONTEXT = Context(
        _cl.Context(
            devices=[device], properties=[(_cl.context_properties.PLATFORM, platform)]
        )
    )
    _DEFAULT_DEVICE = DeviceInterface(
        _DEFAULT_CONTEXT, _DEFAULT_CONTEXT.devices[0].cl_device
    )

    vector_width_single = _DEFAULT_DEVICE.native_vector_width("single")
    vector_width_double = _DEFAULT_DEVICE.native_vector_width("double")

    bempp.api.log(
        f"{_DEFAULT_DEVICE.name}. "
        + f"Device Type: {_DEFAULT_DEVICE.type}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


def get_precision(device_interface):
    """Return precision depending on device."""
    import bempp.api

    if device_interface.type == "cpu":
        return bempp.api.DEVICE_PRECISION_CPU

    if device_interface.type == "gpu":
        return bempp.api.DEVICE_PRECISION_GPU


def show_available_platforms_and_devices():
    """Print available platforms and devices."""
    platforms = _cl.get_platforms()
    for platform_index, platform in enumerate(platforms):
        print(str(platform_index) + ": " + platform.get_info(_cl.platform_info.NAME))
        devices = platform.get_devices()
        for device_index, device in enumerate(devices):
            print(
                4 * " "
                + str(device_index)
                + ": "
                + device.get_info(_cl.device_info.NAME)
            )

def get_context_by_name(identifier):
    """Return context whose name contains the given identifier."""

    platforms = _cl.get_platforms()
    for index, platform in enumerate(platforms):
        if string in platform.name:
            ctx = Context(
                    _cl.Context(
                        dev_type=_cl.device_type.ALL,
                        properties=[(_cl.context_properties.PLATFORM, platform)]
                        )
                    )
            return ctx, index
    raise ValueError(f"No context found whose name contains {identifier}")
                
