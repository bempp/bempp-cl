"""Administrate OpenCL devices."""
import pyopencl as _cl

_DEFAULT_DEVICE = None
_DEFAULT_CONTEXT = None

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
        context = _cl.create_some_context(interactive=False)
        _DEFAULT_CONTEXT = context
        _DEFAULT_DEVICE = context.devices[0]
        bempp.api.log(f"OpenCL Device set to: {_DEFAULT_DEVICE.name}")

    return _DEFAULT_DEVICE

def find_cpu_driver():
    """Find the first available CPU OpenCL driver."""

    for platform in _cl.get_platforms():
        ctx = _cl.Context(
            dev_type=_cl.device_type.ALL,
	    properties=[(_cl.context_properties.PLATFORM, platform)])
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

    if precision == 'single':
        return device.native_vector_width_float
    elif precision == 'double':
        return device.native_vector_width_double
    else:
        raise ValueError("precision must be one of 'single', 'double'.")
