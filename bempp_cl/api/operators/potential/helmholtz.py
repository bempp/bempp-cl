"""Helmholtz potential operators."""
import numpy as _np


def single_layer(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler
    from .modified_helmholtz import single_layer as modified_single_layer

    if _np.real(wavenumber) == 0:
        return modified_single_layer(
            space,
            points,
            wavenumber,
            parameters,
            assembler,
            device_interface,
            precision,
        )

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_single_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def double_layer(
    space,
    points,
    wavenumber,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Helmholtz double-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler
    from .modified_helmholtz import double_layer as modified_double_layer

    if _np.real(wavenumber) == 0:
        return modified_double_layer(
            space,
            points,
            wavenumber,
            parameters,
            assembler,
            device_interface,
            precision,
        )

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "helmholtz_double_layer_potential",  # Identifier
        [_np.real(wavenumber), _np.imag(wavenumber)],  # Options
        "helmholtz_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        True,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )