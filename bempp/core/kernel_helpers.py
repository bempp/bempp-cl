"""Helper routines for running kernels."""
from bempp.core import cl_helpers as _cl_helpers


def run_chunked_kernel(
    main_kernel,
    remainder_kernel,
    device_interface,
    vec_length,
    buffers,
    parameters,
    chunks,
):
    """
    Run a two dimensional kernel with multiple chunks.

    Return the total kernel runtime in ms.
    """

    nchunks1 = len(chunks[0]) - 1
    nchunks2 = len(chunks[1]) - 1

    all_events = []

    for index1 in range(nchunks1):
        for index2 in range(nchunks2):
            dims = (
                chunks[0][index1 + 1] - chunks[0][index1],
                chunks[1][index2 + 1] - chunks[1][index2],
            )
            offsets = (chunks[0][index1], chunks[1][index2])
            events = run_2d_kernel(
                main_kernel,
                remainder_kernel,
                device_interface,
                vec_length,
                buffers,
                parameters,
                dims,
                offsets,
            )
            all_events += events

    _cl_helpers.wait_for_events(all_events)
    return sum([event.runtime() for event in all_events])


def run_2d_kernel(
    main_kernel,
    remainder_kernel,
    device_interface,
    vec_length,
    buffers,
    parameters,
    dims,
    offsets,
):
    """Run a kernel vectorised over the second dimension."""

    if vec_length == 1:
        # Use the workgroup multiple suggested by OpenCL runtime
        # This is always entered on GPUs as vec_length is then 1

        workgroup_multiple = main_kernel.optimal_workgroup_multiple(device_interface)
        workgroup_size = (
            workgroup_multiple * parameters.assembly.dense.workgroup_size_multiple
        )

        # In OpenCL 1.2 the global size always needs to be a multiple
        # of the workgroup size. We run two kernels if this is not possible.
        main, remaining = closest_multiple_to_number(dims[1], workgroup_size)

        events = []
        if main > 0:
            event1 = main_kernel.run(
                device_interface,
                (dims[0], main),
                (1, workgroup_size),
                global_offset=offsets,
                *buffers
            )
            events.append(event1)
        if remaining > 0:
            event2 = main_kernel.run(
                device_interface,
                (dims[0], remaining),
                (1, 1),
                global_offset=(offsets[0], offsets[1] + main),
                *buffers
            )
            events.append(event2)
        return events

    # Split the computation into chunks according to
    # CPU vectorization level

    main_chunk, remaining_chunk = closest_multiple_to_number(dims[1], vec_length)

    events = []

    if main_chunk > 0:
        event1 = main_kernel.run(
            device_interface,
            (dims[0], main_chunk // vec_length),
            (1, 1),
            *buffers,
            global_offset=offsets
        )
        events.append(event1)

    if remaining_chunk > 0:
        event2 = remainder_kernel.run(
            device_interface,
            (dims[0], remaining_chunk),
            (1, 1),
            *buffers,
            global_offset=(offsets[0], offsets[1] + main_chunk)
        )
        events.append(event2)

    return events


def closest_multiple_to_number(number, factor):
    """
    Find the closest multiple of a factor to a given number.

    Returns a tuple (n, r) such that n is a multiple of factor
    and r = number - n < factor.
    """
    r = number % factor  # pylint: disable=C0103

    return (number - r, r)


def get_vectorization_information(device_interface, precision):
    """
    Get correct vectorization information.

    Returns a tuple (extension, vec_length), where
    extension is a string extension to choose the correct
    kernel and vec_length is the vectorization length.
    """
    import bempp.api

    vec_data = {1: ("_novec", 1), 4: ("_vec4", 4), 8: ("_vec8", 8), 16: ("_vec16", 16)}

    # On GPUs vector length is always 1
    if device_interface.type == "gpu":
        return vec_data[1]

    if bempp.api.VECTORIZATION == "auto":
        vector_width = device_interface.native_vector_width(precision)
        if vector_width not in [1, 4, 8, 16]:
            vector_width = 1
        return vec_data[vector_width]
    elif bempp.api.VECTORIZATION == "vec16":
        return vec_data[16]
    elif bempp.api.VECTORIZATION == "vec8":
        return vec_data[8]
    elif bempp.api.VECTORIZATION == "vec4":
        return vec_data[4]
    elif bempp.api.VECTORIZATION == "novec":
        return vec_data[1]

    raise ValueError("Unsupported vectorization option.")
