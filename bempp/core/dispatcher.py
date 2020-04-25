"""Dispatch kernel calls to different implementations."""


def singular_assembler_dispatcher(device_interface, *args):
    """Dispatch the singular assembler to the different implementations."""

    interface_type = device_interface.split("_")[0]

    if interface_type == "opencl":
        from bempp.core.opencl_assemblers import singular_assembler

        singular_assembler(device_interface, *args)

    elif interface_type == "numba":
        from bempp.core.numba_assemblers import singular_assembler

        singular_assembler(device_interface, *args)

    else:

        raise ValueError("Unknown assembler.")
