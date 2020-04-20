"""Common helpers uses for all boundary operators."""
from bempp.api.assembly import assembler as _assembler
from bempp.api.assembly import boundary_operator as _boundary_operator
from bempp.api.assembly import blocked_operator as _blocked_operator
import numpy as _np


# pylint: disable=too-many-arguments
def create_operator(
    identifier,
    domain,
    range_,
    dual_to_range,
    parameters,
    assembler,
    operator_options,
    kernel_type,
    assembly_type,
    device_interface,
    precision,
    is_complex,
    kernel_mode,
):
    """Generic instantiation of operators."""
    from bempp.api.operators import OperatorDescriptor
    import bempp.api

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    if assembler != 'only_singular_part':
        # Add operator for singular part assembly. Needed for certain fast methods.
        singular_part = create_operator(
                identifier, domain, range_, dual_to_range, parameters, 'only_singular_part',
                operator_options, kernel_type, assembly_type, device_interface, precision,
                is_complex)
    else:
        singular_part = None

    assembler = _assembler.AssemblerInterface(
        domain, dual_to_range, assembler, device_interface, precision, parameters
    )

    descriptor = OperatorDescriptor(
        identifier, operator_options, kernel_type, assembly_type, precision, is_complex, singular_part, kernel_mode,
    )
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
    from bempp.api.operators import OperatorDescriptor
    from bempp.api.operators import MultitraceOperatorDescriptor
    import bempp.api

    if device_interface is None:
        device_interface = bempp.api.DEFAULT_DEVICE

    if precision is None:
        precision = bempp.api.DEFAULT_PRECISION

    assembler = _assembler.AssemblerInterface(
        domain, dual_to_range, assembler, device_interface, precision, parameters
    )

    descriptor = MultitraceOperatorDescriptor(
        identifier, operator_options, multitrace_kernel, singular_contribution
    )
    return _blocked_operator.MultitraceOperatorFromAssembler(
        domain, range_, dual_to_range, assembler, descriptor
    )
