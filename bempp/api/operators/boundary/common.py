"""Common helpers uses for all boundary operators."""
from bempp.api.assembly import assembler as _assembler
from bempp.api.assembly import boundary_operator as _boundary_operator
from bempp.api.assembly import blocked_operator as _blocked_operator


# pylint: disable=too-many-arguments
def create_operator(
    identifier,
    domain,
    range_,
    dual_to_range,
    parameters,
    assembler,
    operator_options,
    compute_kernel,
    device_interface,
    precision,
):
    """Generic instantiation of operators."""
    from bempp.api.operators import OperatorDescriptor
    from bempp.api import default_device
    from bempp.api import get_precision

    if device_interface is None:
        device_interface = default_device()

    if precision is None:
        precision = get_precision(device_interface)

    assembler = _assembler.AssemblerInterface(
        domain, dual_to_range, assembler, device_interface, precision, parameters
    )

    descriptor = OperatorDescriptor(identifier, operator_options, compute_kernel)
    return _boundary_operator.BoundaryOperatorWithAssembler(
        domain, range_, dual_to_range, assembler, descriptor
    )


def create_multitrace_operator(
    identifier,
    domain,
    range_,
    dual_to_range,
    parameters,
    assembler,
    operator_options,
    multitrace_kernel,
    singular_contribution,
    device_interface,
    precision,
):
    """Generic instantiation of operators."""
    from bempp.api.operators import MultitraceOperatorDescriptor
    from bempp.api import default_device
    from bempp.api import get_precision

    if device_interface is None:
        device_interface = default_device()

    if precision is None:
        precision = get_precision(device_interface)

    assembler = _assembler.AssemblerInterface(
        domain, dual_to_range, assembler, device_interface, precision, parameters
    )

    descriptor = MultitraceOperatorDescriptor(
        identifier,
        operator_options,
        multitrace_kernel,
        singular_contribution
    )
    return _blocked_operator.MultitraceOperatorFromAssembler(
        domain, range_, dual_to_range, assembler, descriptor
    )
