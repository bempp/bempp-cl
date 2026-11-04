"""Laplace potential operators."""


def single_layer(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
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
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "default_scalar",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        1,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def single_layer_gradient(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_gradient_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "laplace_single_layer_gradient",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def double_layer_gradient(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_gradient_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "laplace_double_layer_gradient",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        3,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )


def single_layer_2nd_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_2nd_deriv_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "laplace_single_layer_2nd_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        6,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
        
def double_layer_2nd_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_2nd_deriv_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "laplace_double_layer_2nd_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        6,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def single_layer_3rd_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_3rd_deriv_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "laplace_single_layer_3rd_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        10,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
        
def double_layer_3rd_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_3rd_deriv_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "laplace_double_layer_3rd_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        10,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def single_layer_4th_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_4th_deriv_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "laplace_single_layer_4th_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        15,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
        
def double_layer_4th_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_double_layer_4th_deriv_potential",  # Identifier
        [],  # Options
        "laplace_double_layer",  # Kernel type
        "laplace_double_layer_4th_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        15,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )

def single_layer_5th_deriv(
    space,
    points,
    parameters=None,
    assembler="dense",
    device_interface=None,
    precision=None,
):
    """Return a Laplace single-layer potential operator."""
    import bempp_cl.api
    from bempp_cl.api.operators import OperatorDescriptor
    from bempp_cl.api.assembly.potential_operator import PotentialOperator
    from bempp_cl.api.assembly.assembler import PotentialAssembler

    if precision is None:
        precision = bempp_cl.api.DEFAULT_PRECISION

    operator_descriptor = OperatorDescriptor(
        "laplace_single_layer_5th_deriv_potential",  # Identifier
        [],  # Options
        "laplace_single_layer",  # Kernel type
        "laplace_single_layer_5th_deriv",  # Assembly type
        precision,  # Precision
        False,  # Is complex
        None,  # Singular part
        21,  # Kernel dimension
    )

    return PotentialOperator(
        PotentialAssembler(
            space, points, operator_descriptor, device_interface, assembler, parameters
        )
    )
