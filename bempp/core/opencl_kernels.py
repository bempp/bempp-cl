"""OpenCL routines."""

import pyopencl as _cl
import os as _os

_CURRENT_PATH = _os.path.dirname(_os.path.realpath(__file__))
_INCLUDE_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/include"))
_KERNEL_PATH = _os.path.abspath(_os.path.join(_CURRENT_PATH, "./sources/kernels"))

_DEFAULT_DEVICE = None
_DEFAULT_CONTEXT = None


def select_cl_kernel(operator_descriptor, mode):
    """Select OpenCL kernel."""

    singular_assemblers = {
        "default_scalar": "evaluate_dense_singular",
        "laplace_hypersingular": "evaluate_dense_laplace_hypersingular_singular",
        "helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_singular",
        "modified_helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_singular",
        "maxwell_electric_field": "evaluate_dense_electric_field_singular",
        "maxwell_magnetic_field": "evaluate_dense_magnetic_field_singular"
    }

    regular_assemblers = {
        "default_scalar": "evaluate_dense_regular",
        "laplace_hypersingular": "evaluate_dense_laplace_hypersingular_regular",
        "helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_regular",
        "modified_helmholtz_hypersingular": "evaluate_dense_helmholtz_hypersingular_regular",
        "maxwell_electric_field": "evaluate_dense_electric_field_regular",
        "maxwell_magnetic_field": "evaluate_dense_magnetic_field_regular"
    }

    potential_assemblers = {
        "default_scalar": "evaluate_scalar_potential",
        "maxwell_electric_field": "evaluate_electric_field_potential",
        "maxwell_magnetic_field": "evaluate_magnetic_field_potential"
    }

    kernels = {
        "laplace_single_layer": "laplace_single_layer",
        "laplace_double_layer": "laplace_double_layer",
        "laplace_adjoint_double_layer": "laplace_adjoint_double_layer",
        "helmholtz_single_layer": "helmholtz_single_layer",
        "helmholtz_double_layer": "helmholtz_double_layer",
        "helmholtz_adjoint_double_layer": "helmholtz_adjoint_double_layer",
        "modified_helmholtz_single_layer": "modified_helmholtz_real_single_layer",
        "modified_helmholtz_double_layer": "modified_helmholtz_real_double_layer",
        "modified_helmholtz_adjoint_double_layer": "modified_helmholtz_real_adjoint_double_layer",
    }

    if mode == "singular":
        return (
            singular_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type],
        )

    elif mode == "regular":
        return (
            regular_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type],
        )
    elif mode == "potential":
        return (
            potential_assemblers[operator_descriptor.assembly_type],
            kernels[operator_descriptor.kernel_type]
            )


def get_kernel_compile_options(options, precision):
    """Create compiler options from parameters and precision."""
    import numbers

    if precision == "single":
        literal = "f"
    else:
        literal = ""

    compile_options = []
    for key, value in options.items():
        if value is None:
            compile_options += ["-D", "{0}".format(key)]
        else:
            if isinstance(value, numbers.Real) and not isinstance(
                value, numbers.Integral
            ):
                value_string = str(value) + literal
            else:
                value_string = str(value)
            compile_options += ["-D", "{0}={1}".format(key, value_string)]

    compile_options += ["-I", _INCLUDE_PATH]

    # Add precision flag

    if precision == "single":
        val = 0
    else:
        val = 1

    compile_options.append("-DPRECISION={0}".format(val))

    return compile_options


def build_program(assembly_function, options, precision):
    """Build the kernel and return it."""

    file_name = assembly_function + ".cl"
    kernel_file = _os.path.join(_KERNEL_PATH, file_name)

    kernel_string = open(kernel_file).read()
    kernel_options = get_kernel_compile_options(options, precision)

    return (
        _cl.Program(default_context(), kernel_string)
        .build(options=kernel_options)
        .kernel_function
    )


def get_kernel_from_operator_descriptor(
    operator_descriptor, options, mode, force_novec=False
):
    """Return compiled kernel from operator descriptor."""

    precision = operator_descriptor.precision
    assembly_function, kernel_name = select_cl_kernel(operator_descriptor, mode=mode)

    vec_length = get_vector_width(precision)
    vec_string = get_vec_string(precision)

    if not mode == "singular":
        if force_novec or vec_length == 1:
            assembly_function += "_novec"
        else:
            assembly_function += "_vec"
    options["KERNEL_FUNCTION"] = kernel_name
    options["VEC_LENGTH"] = vec_length
    options["VEC_STRING"] = vec_string
    return build_program(assembly_function, options, precision)

def get_kernel_from_name(
    name, options, precision="double"
):
    """Return compiled kernel from name."""

    vec_length = get_vector_width(precision)
    vec_string = get_vec_string(precision)

    options["VEC_LENGTH"] = vec_length
    options["VEC_STRING"] = vec_string
    return build_program(name, options, precision)


def get_vector_width(precision):
    """Return vector width."""
    import bempp.api

    mode_to_length = {"novec": 1, "vec4": 4, "vec8": 8, "vec16": 16}

    if bempp.api.VECTORIZATION_MODE == "auto":
        return get_native_vector_width(default_device(), precision)
    else:
        return mode_to_length[bempp.api.VECTORIZATION_MODE]


def get_vec_string(precision):
    """Return vectorisation string."""
    import bempp.api

    vec_strings = {1: "novec", 4: "vec4", 8: "vec8", 16: "vec16"}

    return vec_strings[get_vector_width(precision)]


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
                bempp.api.log(f"OpenCL Device set to: {_DEFAULT_DEVICE.name}")
                return _DEFAULT_DEVICE
        context = _cl.create_some_context(interactive=False)
        _DEFAULT_CONTEXT = context
        _DEFAULT_DEVICE = context.devices[0]
        bempp.api.log(f"OpenCL Device set to: {_DEFAULT_DEVICE.name}")

    return _DEFAULT_DEVICE


def default_context():
    """Return default context."""

    if _DEFAULT_CONTEXT is None:
        default_device()

    return _DEFAULT_CONTEXT


def find_cpu_driver():
    """Find the first available CPU OpenCL driver."""

    for platform in _cl.get_platforms():
        ctx = _cl.Context(
            dev_type=_cl.device_type.ALL,
            properties=[(_cl.context_properties.PLATFORM, platform)],
        )
        for device in ctx.devices:
            if device.type == _cl.device_type.CPU:
                return ctx, device
    return None


def set_default_device(platform_index, device_index):
    """Set the default device."""
    import bempp.api

    # pylint: disable=W0603
    global _DEFAULT_DEVICE
    global _DEFAULT_CONTEXT

    platform = _cl.get_platforms()[platform_index]
    device = platform.get_devices()[device_index]
    _DEFAULT_CONTEXT = _cl.Context(
        devices=[device], properties=[(_cl.context_properties.PLATFORM, platform)]
    )
    _DEFAULT_DEVICE = _DEFAULT_CONTEXT.devices[0]

    vector_width_single = _DEFAULT_DEVICE.native_vector_width_float
    vector_width_double = _DEFAULT_DEVICE.native_vector_width_double

    bempp.api.log(
        f"Default device: {_DEFAULT_DEVICE.name}. "
        + f"Device Type: {_DEFAULT_DEVICE.type}. "
        + f"Native vector width: {vector_width_single} (single) / "
        + f"{vector_width_double} (double)."
    )


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


def get_native_vector_width(device, precision):
    """Get default vector width for device."""

    if precision == "single":
        return device.native_vector_width_float
    elif precision == "double":
        return device.native_vector_width_double
    else:
        raise ValueError("precision must be one of 'single', 'double'.")
